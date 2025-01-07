package com.example.achats.service;

import com.example.achats.model.Fournisseur;
import com.example.achats.repository.FournisseurRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class FournisseurService {
    @Autowired
    private FournisseurRepository repository;

public void update(Long id, Fournisseur fournisseur) {
    fournisseur.setId(id);
    repository.save(fournisseur);
}

public void ajouter(Fournisseur fournisseur) {
        repository.save(fournisseur);
    }

    public Optional<Fournisseur> rechercheParId(Long id) {
        return repository.findById(id);
    }

    public List<Fournisseur> rechercheTout() {
        return repository.findAll();
    }

    public void supprimer(Long id) {
        repository.deleteById(id);
    }
}
